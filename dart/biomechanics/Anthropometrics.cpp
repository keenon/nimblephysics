#include "dart/biomechanics/Anthropometrics.hpp"

#include <algorithm>

#include "dart/math/FiniteDifference.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/XmlHelpers.hpp"

namespace dart {

namespace biomechanics {

//==============================================================================
AnthroMetric::AnthroMetric(
    std::string name,
    Eigen::VectorXs bodyPose,
    std::string bodyA,
    Eigen::Vector3s offsetA,
    std::string bodyB,
    Eigen::Vector3s offsetB,
    Eigen::Vector3s axis)
  : name(name),
    bodyPose(bodyPose),
    bodyA(bodyA),
    offsetA(offsetA),
    bodyB(bodyB),
    offsetB(offsetB),
    axis(axis)
{
}

//==============================================================================
Anthropometrics::Anthropometrics()
{
}

//==============================================================================
std::shared_ptr<Anthropometrics> Anthropometrics::loadFromFile(
    const common::Uri& uri, const common::ResourceRetrieverPtr& retrieverOrNull)
{
  common::ResourceRetrieverPtr retriever = nullptr;
  if (retrieverOrNull)
  {
    retriever = retrieverOrNull;
  }
  else
  {
    auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
    newRetriever->addSchemaRetriever(
        "file", std::make_shared<common::LocalResourceRetriever>());
    newRetriever->addSchemaRetriever(
        "dart", utils::DartResourceRetriever::create());
    retriever = newRetriever;
  }

  std::shared_ptr<Anthropometrics> result = std::make_shared<Anthropometrics>();

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument xmlFile;
  try
  {
    utils::openXMLFile(xmlFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return result;
  }

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement = xmlFile.FirstChildElement("Metrics");
  if (docElement == nullptr)
  {
    dterr << "Anthropometrics file[" << uri.toString()
          << "] does not contain <Metrics> as the root element.\n";
    return result;
  }
  tinyxml2::XMLElement* metricElement = docElement->FirstChildElement("Metric");
  if (metricElement == nullptr)
  {
    dterr << "Anthropometrics file[" << uri.toString()
          << "] does not contain <Metric> as the child of the root "
             "<Metrics> element.\n";
    return result;
  }

  while (metricElement != nullptr)
  {
    tinyxml2::XMLElement* nameElement
        = metricElement->FirstChildElement("Name");
    std::string name = nameElement->GetText();

    tinyxml2::XMLElement* markerAElement
        = metricElement->FirstChildElement("MarkerA");
    std::string bodyA = utils::getValueString(markerAElement, "BodyNode");
    Eigen::Vector3s offsetA = utils::getValueVector3s(markerAElement, "Offset");

    tinyxml2::XMLElement* markerBElement
        = metricElement->FirstChildElement("MarkerB");
    std::string bodyB = utils::getValueString(markerBElement, "BodyNode");
    Eigen::Vector3s offsetB = utils::getValueVector3s(markerBElement, "Offset");

    Eigen::Vector3s axis
        = utils::getValueVector3s(metricElement, "MeasureAlongAxis");

    Eigen::VectorXs bodyPose = Eigen::VectorXs::Zero(0);
    if (metricElement->FirstChildElement("BodyPose"))
    {
      bodyPose = utils::getValueVectorXs(metricElement, "BodyPose");
    }

    result->addMetric(name, bodyPose, bodyA, offsetA, bodyB, offsetB, axis);

    metricElement = metricElement->NextSiblingElement("Metric");
  }

  return result;
}

//==============================================================================
void Anthropometrics::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    std::shared_ptr<dynamics::Skeleton> skel)
{
  std::vector<Eigen::Vector3s> colors;
  colors.push_back(Eigen::Vector3s(255, 0, 0));
  colors.push_back(Eigen::Vector3s(0, 255, 0));
  colors.push_back(Eigen::Vector3s(0, 0, 255));
  colors.push_back(Eigen::Vector3s(255, 255, 0));
  colors.push_back(Eigen::Vector3s(255, 0, 255));
  colors.push_back(Eigen::Vector3s(0, 255, 255));
  colors.push_back(Eigen::Vector3s(127, 64, 64));
  colors.push_back(Eigen::Vector3s(242, 137, 121));
  colors.push_back(Eigen::Vector3s(153, 69, 38));
  colors.push_back(Eigen::Vector3s(242, 97, 0));
  colors.push_back(Eigen::Vector3s(64, 26, 0));
  colors.push_back(Eigen::Vector3s(230, 195, 172));
  colors.push_back(Eigen::Vector3s(140, 91, 35));
  colors.push_back(Eigen::Vector3s(229, 176, 115));
  colors.push_back(Eigen::Vector3s(64, 49, 32));
  colors.push_back(Eigen::Vector3s(255, 204, 0));
  colors.push_back(Eigen::Vector3s(140, 119, 35));
  colors.push_back(Eigen::Vector3s(222, 230, 115));
  colors.push_back(Eigen::Vector3s(170, 255, 0));
  colors.push_back(Eigen::Vector3s(85, 102, 51));
  colors.push_back(Eigen::Vector3s(91, 140, 35));
  colors.push_back(Eigen::Vector3s(206, 242, 182));
  colors.push_back(Eigen::Vector3s(0, 64, 17));
  colors.push_back(Eigen::Vector3s(61, 242, 109));
  colors.push_back(Eigen::Vector3s(54, 217, 163));
  colors.push_back(Eigen::Vector3s(70, 140, 126));
  colors.push_back(Eigen::Vector3s(67, 89, 85));
  colors.push_back(Eigen::Vector3s(128, 247, 255));
  colors.push_back(Eigen::Vector3s(0, 184, 230));
  colors.push_back(Eigen::Vector3s(0, 102, 153));
  colors.push_back(Eigen::Vector3s(0, 43, 64));
  colors.push_back(Eigen::Vector3s(182, 222, 242));
  colors.push_back(Eigen::Vector3s(115, 145, 230));
  colors.push_back(Eigen::Vector3s(105, 115, 140));
  colors.push_back(Eigen::Vector3s(0, 0, 128));
  colors.push_back(Eigen::Vector3s(13, 13, 51));
  colors.push_back(Eigen::Vector3s(85, 61, 242));
  colors.push_back(Eigen::Vector3s(79, 70, 140));
  colors.push_back(Eigen::Vector3s(202, 121, 242));
  colors.push_back(Eigen::Vector3s(75, 57, 77));
  colors.push_back(Eigen::Vector3s(230, 0, 214));
  colors.push_back(Eigen::Vector3s(255, 191, 251));
  colors.push_back(Eigen::Vector3s(128, 32, 96));
  colors.push_back(Eigen::Vector3s(140, 105, 129));
  colors.push_back(Eigen::Vector3s(242, 0, 129));
  colors.push_back(Eigen::Vector3s(64, 0, 26));
  colors.push_back(Eigen::Vector3s(229, 0, 31));
  colors.push_back(Eigen::Vector3s(255, 191, 200));

  server->deleteObjectsByPrefix("anthro_metric_");
  for (int i = 0; i < mMetrics.size(); i++)
  {
    AnthroMetric& metric = mMetrics.at(i);
    Eigen::Vector4s color = Eigen::Vector4s(
        colors[i](0) / 255.0, colors[i](1) / 255.0, colors[i](2) / 255.0, 1.0);
    Eigen::Vector4s colorTransparent = Eigen::Vector4s(
        colors[i](0) / 255.0, colors[i](1) / 255.0, colors[i](2) / 255.0, 0.4);

    if (skel->getBodyNode(metric.bodyA) == nullptr
        || skel->getBodyNode(metric.bodyB) == nullptr)
      continue;

    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
    markers.push_back(std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
        skel->getBodyNode(metric.bodyA), metric.offsetA));
    markers.push_back(std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
        skel->getBodyNode(metric.bodyB), metric.offsetB));
    Eigen::VectorXs worldSpace = skel->getMarkerWorldPositions(markers);
    Eigen::Vector3s markerA = worldSpace.head<3>();
    Eigen::Vector3s markerB = worldSpace.tail<3>();

    // Create spheres at the endpoints
    server->createSphere(
        "anthro_metric_a_" + metric.name, 0.01, markerA, colorTransparent);
    server->createSphere(
        "anthro_metric_b_" + metric.name, 0.01, markerB, colorTransparent);

    // Draw lines for each metric

    // If the axis is zero, then this is a raw distance measurement in world
    // space
    if (metric.axis == Eigen::Vector3s::Zero())
    {
      std::vector<Eigen::Vector3s> points;
      points.push_back(markerA);
      points.push_back(markerB);
      server->createLine("anthro_metric_" + metric.name, points, color);
    }
    // If the axis is non-zero, then this is a distance measurement along the
    // axis.
    else
    {
      Eigen::Vector3s diff = markerA - markerB;
      Eigen::Vector3s diffAlongAxis = diff.dot(metric.axis) * metric.axis;
      Eigen::Vector3s diffPerpendicular = diff - diffAlongAxis;

      if (diffPerpendicular.norm() > 0)
      {
        Eigen::Matrix3s R = Eigen::Matrix3s::Zero();
        R.col(0) = diffPerpendicular.normalized();
        R.col(1) = metric.axis.normalized();
        R.col(2) = R.col(0).cross(R.col(1));

        Eigen::Vector3s euler = math::matrixToEulerXYZ(R);
        Eigen::Vector3s size = Eigen::Vector3s(
            diffPerpendicular.norm(), 0.005, diffPerpendicular.norm());
        server->createBox(
            "anthro_metric_top_box_" + metric.name,
            size,
            markerA,
            euler,
            colorTransparent);

        server->createBox(
            "anthro_metric_bottom_box_" + metric.name,
            size,
            markerB,
            euler,
            colorTransparent);
      }

      // Jump to the middle, then shoot across the perpendicular section
      std::vector<Eigen::Vector3s> points;
      points.push_back(markerA);
      points.push_back(markerA - diffPerpendicular / 2);
      points.push_back(markerB + diffPerpendicular / 2);
      points.push_back(markerB);
      server->createLine("anthro_metric_" + metric.name, points, color);
    }
  }
}

//==============================================================================
void Anthropometrics::addMetric(
    std::string name,
    Eigen::VectorXs bodyPose,
    std::string bodyA,
    Eigen::Vector3s offsetA,
    std::string bodyB,
    Eigen::Vector3s offsetB,
    Eigen::Vector3s axis)
{
  mMetrics.emplace_back(name, bodyPose, bodyA, offsetA, bodyB, offsetB, axis);
}

//==============================================================================
std::vector<std::string> Anthropometrics::getMetricNames()
{
  std::vector<std::string> names;
  for (auto m : mMetrics)
  {
    if (std::find(names.begin(), names.end(), m.name) == names.end())
    {
      names.push_back(m.name);
    }
  }
  return names;
}

//==============================================================================
void Anthropometrics::setDistribution(
    std::shared_ptr<math::MultivariateGaussian> dist)
{
  mDist = dist;
}

//==============================================================================
std::shared_ptr<math::MultivariateGaussian> Anthropometrics::getDistribution()
{
  return mDist;
}

//==============================================================================
std::shared_ptr<Anthropometrics> Anthropometrics::condition(
    const std::map<std::string, s_t>& observedValues)
{
  std::shared_ptr<Anthropometrics> conditioned
      = std::make_shared<Anthropometrics>();
  for (auto metric : mMetrics)
  {
    // Only copy over metrics we're not conditioning out already
    if (observedValues.count(metric.name) == 0)
    {
      conditioned->addMetric(
          metric.name,
          metric.bodyPose,
          metric.bodyA,
          metric.offsetA,
          metric.bodyB,
          metric.offsetB,
          metric.axis);
    }
  }
  conditioned->setDistribution(mDist->condition(observedValues));
  return conditioned;
}

//==============================================================================
void Anthropometrics::setSkelToMetricPose(
    std::shared_ptr<dynamics::Skeleton> skel, const AnthroMetric& metric)
{
  if (metric.bodyPose.size() != 0)
  {
    if (skel->getNumDofs() == metric.bodyPose.size())
    {
      if (skel->getPositions() != metric.bodyPose)
      {
        skel->setPositions(metric.bodyPose);
      }
    }
    else
    {
      /*
      std::cout << "WARNING: Anthropometric \"" << metric.name
                << "\" specifies a BodyPose with size "
                << metric.bodyPose.size() << ", but skeleton has "
                << skel->getNumDofs() << " DOFs." << std::endl;
      */

      // Trim/pad with zeros
      Eigen::VectorXs padded = Eigen::VectorXs::Zero(skel->getNumDofs());
      if (padded.size() > metric.bodyPose.size())
      {
        padded.segment(0, metric.bodyPose.size()) = metric.bodyPose;
      }
      else
      {
        padded = metric.bodyPose.segment(0, padded.size());
      }

      // Now set positions
      if (skel->getPositions() != padded)
      {
        skel->setPositions(padded);
      }
    }
  }
}

//==============================================================================
std::map<std::string, s_t> Anthropometrics::measure(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  std::map<std::string, s_t> result;
  Eigen::VectorXs originalPos = skel->getPositions();
  for (AnthroMetric& metric : mMetrics)
  {
    setSkelToMetricPose(skel, metric);
    if (skel->getBodyNode(metric.bodyA) == nullptr
        || skel->getBodyNode(metric.bodyB) == nullptr)
    {
      if (result.count(metric.name) == 0)
      {
        result[metric.name] = mDist->getMean(metric.name);
      }
    }
    else
    {
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyA), metric.offsetA);
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyB), metric.offsetB);

      if (metric.axis == Eigen::Vector3s::Zero())
      {
        result[metric.name] = skel->getDistanceInWorldSpace(markerA, markerB);
      }
      else
      {
        result[metric.name]
            = skel->getDistanceAlongAxis(markerA, markerB, metric.axis);
      }
    }
  }
  skel->setPositions(originalPos);
  return result;
}

//==============================================================================
s_t Anthropometrics::getPDF(std::shared_ptr<dynamics::Skeleton> skel)
{
  if (!mDist)
    return 0.0;
  return mDist->computePDF(mDist->convertFromMap(measure(skel)));
}

//==============================================================================
s_t Anthropometrics::getLogPDF(
    std::shared_ptr<dynamics::Skeleton> skel, bool normalized)
{
  if (!mDist)
    return 0.0;
  return mDist->computeLogPDF(mDist->convertFromMap(measure(skel)), normalized);
}

//==============================================================================
Eigen::VectorXs Anthropometrics::getGradientOfLogPDFWrtBodyScales(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(skel->getNumBodyNodes() * 3);
  if (!mDist)
    return grad;

  std::map<std::string, s_t> gradMap = mDist->convertToMap(
      mDist->computeLogPDFGrad(mDist->convertFromMap(measure(skel))));

  Eigen::VectorXs originalPos = skel->getPositions();
  for (AnthroMetric& metric : mMetrics)
  {
    setSkelToMetricPose(skel, metric);

    if (skel->getBodyNode(metric.bodyA) != nullptr
        && skel->getBodyNode(metric.bodyB) != nullptr)
    {
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyA), metric.offsetA);
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyB), metric.offsetB);
      if (metric.axis == Eigen::Vector3s::Zero())
      {
        grad += gradMap[metric.name]
                * skel->getGradientOfDistanceWrtBodyScales(markerA, markerB);
      }
      else
      {
        grad += gradMap[metric.name]
                * skel->getGradientOfDistanceAlongAxisWrtBodyScales(
                    markerA, markerB, metric.axis);
      }
    }
  }
  skel->setPositions(originalPos);

  return grad;
}

//==============================================================================
Eigen::VectorXs Anthropometrics::finiteDifferenceGradientOfLogPDFWrtBodyScales(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(skel->getNumBodyNodes() * 3);

  Eigen::VectorXs originalScales = skel->getBodyScales();

  s_t eps = 1e-3;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs tweaked = originalScales;
        tweaked(i) += eps;
        skel->setBodyScales(tweaked);
        out = getLogPDF(skel);
        return true;
      },
      result,
      eps,
      true);

  skel->setBodyScales(originalScales);

  return result;
}

//==============================================================================
Eigen::VectorXs Anthropometrics::getGradientOfLogPDFWrtGroupScales(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(skel->getGroupScaleDim());
  if (!mDist)
    return grad;

  std::map<std::string, s_t> gradMap = mDist->convertToMap(
      mDist->computeLogPDFGrad(mDist->convertFromMap(measure(skel))));

  Eigen::VectorXs originalPos = skel->getPositions();
  for (AnthroMetric& metric : mMetrics)
  {
    setSkelToMetricPose(skel, metric);

    if (skel->getBodyNode(metric.bodyA) != nullptr
        && skel->getBodyNode(metric.bodyB) != nullptr)
    {
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyA), metric.offsetA);
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB
          = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
              skel->getBodyNode(metric.bodyB), metric.offsetB);
      if (metric.axis == Eigen::Vector3s::Zero())
      {
        grad += gradMap[metric.name]
                * skel->getGradientOfDistanceWrtGroupScales(markerA, markerB);
      }
      else
      {
        grad += gradMap[metric.name]
                * skel->getGradientOfDistanceAlongAxisWrtGroupScales(
                    markerA, markerB, metric.axis);
      }
    }
  }
  skel->setPositions(originalPos);

  return grad;
}

//==============================================================================
Eigen::VectorXs Anthropometrics::finiteDifferenceGradientOfLogPDFWrtGroupScales(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(skel->getGroupScaleDim());

  Eigen::VectorXs originalScales = skel->getGroupScales();

  s_t eps = 1e-3;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs tweaked = originalScales;
        tweaked(i) += eps;
        skel->setGroupScales(tweaked);
        out = getLogPDF(skel);
        return true;
      },
      result,
      eps,
      true);

  skel->setGroupScales(originalScales);

  return result;
}

} // namespace biomechanics
} // namespace dart