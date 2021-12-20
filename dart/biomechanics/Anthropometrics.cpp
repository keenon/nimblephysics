#include "dart/biomechanics/Anthropometrics.hpp"

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
Anthropometrics Anthropometrics::loadFromFile(
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

  Anthropometrics result = Anthropometrics();

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

    result.addMetric(name, bodyPose, bodyA, offsetA, bodyB, offsetB, axis);

    metricElement = metricElement->NextSiblingElement("Metric");
  }

  return result;
}

//==============================================================================
void Anthropometrics::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    std::shared_ptr<dynamics::Skeleton> skel)
{
  server->deleteObjectsByPrefix("anthro_metric_");
  for (AnthroMetric& metric : mMetrics)
  {
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
    markers.push_back(std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
        skel->getBodyNode(metric.bodyA), metric.offsetA));
    markers.push_back(std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
        skel->getBodyNode(metric.bodyB), metric.offsetB));
    Eigen::VectorXs worldSpace = skel->getMarkerWorldPositions(markers);
    Eigen::Vector3s markerA = worldSpace.head<3>();
    Eigen::Vector3s markerB = worldSpace.tail<3>();

    std::vector<Eigen::Vector3s> points;
    points.push_back(markerA);
    points.push_back(markerB);
    server->createLine("anthro_metric_" + metric.name, points);
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
    names.push_back(m.name);
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
      std::cout << "WARNING: Anthropometric \"" << metric.name
                << "\" specifies a BodyPose with size "
                << metric.bodyPose.size() << ", but skeleton has "
                << skel->getNumDofs() << " DOFs." << std::endl;
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