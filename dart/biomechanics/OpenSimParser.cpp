#include "dart/biomechanics/OpenSimParser.hpp"

#include <unordered_map>
#include <vector>

#include "dart/common/Uri.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/TranslationalJoint2D.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/PolynomialFunction.hpp"
#include "dart/math/SimmSpline.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/XmlHelpers.hpp"

using namespace std;

// Source: https://stackoverflow.com/a/145309/13177487
#include <stdio.h> /* defines FILENAME_MAX */
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

namespace dart {
using namespace utils;
namespace biomechanics {

//==============================================================================
OpenSimFile::OpenSimFile(
    dynamics::SkeletonPtr skeleton, dynamics::MarkerMap markersMap)
  : skeleton(skeleton), markersMap(markersMap)
{
}

//==============================================================================
OpenSimFile::OpenSimFile()
{
}

//==============================================================================
common::ResourceRetrieverPtr ensureRetriever(
    const common::ResourceRetrieverPtr& _retriever)
{
  if (_retriever)
  {
    return _retriever;
  }
  else
  {
    auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
    newRetriever->addSchemaRetriever(
        "file", std::make_shared<common::LocalResourceRetriever>());
    newRetriever->addSchemaRetriever("dart", DartResourceRetriever::create());
    return newRetriever;
  }
}

//==============================================================================
Eigen::Vector2s readVec2(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector2s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  return vec;
}

Eigen::Vector3s readVec3(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector3s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  vec(2) = strtod(q, &q);
  return vec;
}

Eigen::Vector6s readVec6(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector6s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  vec(2) = strtod(q, &q);
  vec(3) = strtod(q, &q);
  vec(4) = strtod(q, &q);
  vec(5) = strtod(q, &q);
  return vec;
}

std::vector<s_t> readVecX(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q = const_cast<char*>(c);
  std::vector<s_t> values;
  while (q != nullptr && *q != '\0')
  {
    s_t value = strtod(q, &q);
    values.push_back(value);
  }
  return values;
}

dynamics::EulerJoint::AxisOrder getAxisOrder(
    std::vector<Eigen::Vector3s> axisList)
{
  if (axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitZ())
  {
    return dynamics::EulerJoint::AxisOrder::XYZ;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitX())
  {
    return dynamics::EulerJoint::AxisOrder::ZYX;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return dynamics::EulerJoint::AxisOrder::ZXY;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return dynamics::EulerJoint::AxisOrder::XZY;
  }
  assert(false);
  // don't break the build when building as prod
  return dynamics::EulerJoint::AxisOrder::XYZ;
}

Eigen::Vector3s getAxisFlips(std::vector<Eigen::Vector3s> axisList)
{
  Eigen::Vector3s vec;
  vec(0) = (axisList[0] == -Eigen::Vector3s::UnitX()
            || axisList[0] == -Eigen::Vector3s::UnitY()
            || axisList[0] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  vec(1) = (axisList[1] == -Eigen::Vector3s::UnitX()
            || axisList[1] == -Eigen::Vector3s::UnitY()
            || axisList[1] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  vec(2) = (axisList[2] == -Eigen::Vector3s::UnitX()
            || axisList[2] == -Eigen::Vector3s::UnitY()
            || axisList[2] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  return vec;
}

//==============================================================================
OpenSimFile OpenSimParser::parseOsim(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument osimFile;
  try
  {
    openXMLFile(osimFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return null_file;
  }

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = osimFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (jointSet != nullptr)
  {
    // This is the older format, where JointSet specifies the joints separately
    // from the body hierarchy
    return readOsim30(uri, docElement, retriever);
  }
  else
  {
    // This is the newer format, where Joints are specified as childen of Bodies
    return readOsim40(uri, docElement, retriever);
  }
}

//==============================================================================
/// This grabs the marker trajectories from a TRC file
OpenSimTRC OpenSimParser::loadTRC(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimTRC result;
  const std::string content = retriever->readAll(uri);
  double unitsMultiplier = 1.0;

  std::vector<std::string> markerNames;
  Eigen::Vector3s markerSwapSpace = Eigen::Vector3s::Zero();

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    std::map<std::string, Eigen::Vector3s> markerPositions;
    double timestamp = 0.0;

    int tokenNumber = 0;
    std::string whitespace = " \t";
    auto tokenStart = line.find_first_not_of(whitespace);
    while (tokenStart != std::string::npos)
    {
      auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
      std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

      /////////////////////////////////////////////////////////
      // Process the token, given tokenNumber and lineNumber

      if (lineNumber == 2)
      {
        if (tokenNumber == 0)
        { // DataRate
          // timestep = 1.0 / atof(token.c_str());
        }
        if (tokenNumber == 4)
        { // Units
          if (token == "m")
            unitsMultiplier = 1.0;
          else if (token == "mm")
            unitsMultiplier = 1.0 / 1000;
          else
          {
            std::cout << "Unknown length units in .trc file: \"" << token
                      << "\"" << std::endl;
          }
        }
      }
      else if (lineNumber == 3 && tokenNumber > 1)
      {
        markerNames.push_back(token);
      }
      else if (lineNumber > 5)
      {
        if (tokenNumber == 1)
        {
          timestamp = atof(token.c_str());
        }
        else if (tokenNumber > 1)
        {
          int offset
              = tokenNumber - 2; // first two cols are "frame #" and "time"
          int markerNumber = (int)floor((double)offset / 3);
          int axisNumber = offset - (markerNumber * 3);
          markerSwapSpace(axisNumber) = atof(token.c_str()) * unitsMultiplier;
          if (axisNumber == 2)
          {
            if (!markerSwapSpace.hasNaN())
            {
              markerPositions[markerNames[markerNumber]]
                  = Eigen::Vector3s(markerSwapSpace);
            }
          }
        }
      }

      /////////////////////////////////////////////////////////

      tokenNumber++;
      if (tokenEnd == std::string::npos)
      {
        break;
      }
      tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
    }

    if (lineNumber > 5)
    {
      result.markerTimesteps.push_back(markerPositions);
      result.timestamps.push_back(timestamp);
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
    lineNumber++;
  }

  // Translate into a "lines" format, where each marker gets a full trajectory
  for (int i = 0; i < result.markerTimesteps.size(); i++)
  {
    // TODO: this will result in a bug if some timesteps are missing marker
    // observations
    for (auto pair : result.markerTimesteps[i])
    {
      result.markerLines[pair.first].push_back(pair.second);
    }
  }

  return result;
}

//==============================================================================
/// This grabs the joint angles from a *.mot file
OpenSimMot OpenSimParser::loadMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const common::Uri& uri,
    int downsampleByFactor,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimTRC result;
  const std::string content = retriever->readAll(uri);
  std::vector<int> columnToDof;
  std::vector<bool> rotationalDof;

  std::vector<Eigen::VectorXs> poses;
  std::vector<double> timestamps;

  bool inHeader = true;
  bool inDegrees = false;

  int downsampleClock = 0;
  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    // Trim '\r', in case this file was saved on a Windows machine
    if (line.size() > 0 && line[line.size() - 1] == '\r')
    {
      line = line.substr(0, line.size() - 1);
    }

    if (inHeader)
    {
      std::string ENDHEADER = "endheader";
      if (line.size() >= ENDHEADER.size()
          && line.substr(0, ENDHEADER.size()) == ENDHEADER)
      {
        inHeader = false;
      }
      auto tokenEnd = line.find("=");
      if (tokenEnd != std::string::npos)
      {
        std::string variable = line.substr(0, tokenEnd);
        std::string value
            = line.substr(tokenEnd + 1, line.size() - tokenEnd - 1);
        if (variable == "inDegrees")
        {
          inDegrees = (value == "yes");
        }
      }
    }
    else
    {
      int tokenNumber = 0;
      std::string whitespace = " \t";
      auto tokenStart = line.find_first_not_of(whitespace);
      Eigen::VectorXs pose = Eigen::VectorXs::Zero(skel->getNumDofs());
      double timestamp = 0.0;
      while (tokenStart != std::string::npos)
      {
        auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
        std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

        /////////////////////////////////////////////////////////
        // Process the token, given tokenNumber and lineNumber

        if (lineNumber == 0)
        {
          if (tokenNumber > 0)
          {
            // This means we're on the row defining the names of the joints
            // we're recording positions of
            dynamics::DegreeOfFreedom* dof = skel->getDof(token);
            if (dof != nullptr)
            {
              columnToDof.push_back(dof->getIndexInSkeleton());
            }
            else
            {
              columnToDof.push_back(-1);
            }
            dynamics::Joint* joint = dof->getJoint();
            bool isRotationalJoint = true;
            if (joint->getType()
                    == dynamics::TranslationalJoint2D::getStaticType()
                || joint->getType()
                       == dynamics::TranslationalJoint::getStaticType()
                || joint->getType()
                       == dynamics::PrismaticJoint::getStaticType())
            {
              isRotationalJoint = false;
            }
            if (joint->getType() == dynamics::EulerFreeJoint::getStaticType()
                && dof->getIndexInJoint() >= 3)
            {
              isRotationalJoint = false;
            }
            rotationalDof.push_back(isRotationalJoint);
          }
        }
        else
        {
          double value = atof(token.c_str());
          if (tokenNumber == 0)
          {
            timestamp = value;
          }
          else
          {
            int dofIndex = columnToDof[tokenNumber - 1];
            if (dofIndex != -1)
            {
              bool isRotationalJoint = rotationalDof[tokenNumber - 1];
              if (inDegrees && isRotationalJoint)
              {
                value *= M_PI / 180.0;
              }
              pose(dofIndex) = value;
            }
          }
        }

        /////////////////////////////////////////////////////////

        tokenNumber++;
        if (tokenEnd == std::string::npos)
        {
          break;
        }
        tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
      }

      if (lineNumber > 0)
      {
        downsampleClock--;
        if (downsampleClock <= 0)
        {
          downsampleClock = downsampleByFactor;
          poses.push_back(pose);
          timestamps.push_back(timestamp);
        }
      }
      lineNumber++;
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }
  Eigen::MatrixXs posesMatrix
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), poses.size());
  for (int i = 0; i < poses.size(); i++)
  {
    posesMatrix.col(i) = poses[i];
  }
  OpenSimMot mot;
  mot.poses = posesMatrix;
  mot.timestamps = timestamps;

  return mot;
}

//==============================================================================
/// This grabs the GRF forces from a *.mot file
OpenSimGRF OpenSimParser::loadGRF(
    const common::Uri& uri,
    int downsampleByFactor,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimGRF result;
  const std::string content = retriever->readAll(uri);

  bool inHeader = true;

  std::vector<int> colToPlate;
  std::vector<int> colToCOP;
  std::vector<int> colToWrench;
  int numPlates = 0;

  std::vector<s_t> timestamps;
  std::vector<std::vector<Eigen::Vector3s>> copRows;
  std::vector<std::vector<Eigen::Vector6s>> wrenchRows;

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    // Trim '\r', in case this file was saved on a Windows machine
    if (line.size() > 0 && line[line.size() - 1] == '\r')
    {
      line = line.substr(0, line.size() - 1);
    }

    if (inHeader)
    {
      std::string ENDHEADER = "endheader";
      if (line.size() >= ENDHEADER.size()
          && line.substr(0, ENDHEADER.size()) == ENDHEADER)
      {
        inHeader = false;
      }
      auto tokenEnd = line.find("=");
      if (tokenEnd != std::string::npos)
      {
        std::string variable = line.substr(0, tokenEnd);
        std::string value
            = line.substr(tokenEnd + 1, line.size() - tokenEnd - 1);
        // Currently we don't read anything from the variables
        (void)variable;
        (void)value;
      }
    }
    else
    {
      int tokenNumber = 0;
      std::string whitespace = " \t";
      auto tokenStart = line.find_first_not_of(whitespace);

      double timestamp = 0.0;
      std::vector<Eigen::Vector3s> cops;
      std::vector<Eigen::Vector6s> wrenches;

      for (int i = 0; i < numPlates; i++)
      {
        wrenches.push_back(Eigen::Vector6s::Zero());
        cops.push_back(Eigen::Vector3s::Zero());
      }

      while (tokenStart != std::string::npos)
      {
        auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
        std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

        /////////////////////////////////////////////////////////
        // Process the token, given tokenNumber and lineNumber

        if (lineNumber == 0)
        {
          int plate = -1;
          int cop = -1;
          int wrench = -1;

          if (token != "time")
          {
            // Default to plate 0
            plate = 0;
          }

          // Find plate number
          for (int p = 1; p < 10; p++)
          {
            if (token.find(std::to_string(p)) != std::string::npos)
            {
              plate = p - 1;
            }
          }
          // It's pretty common to do R and L plates, instead of numbered plates
          if (token.find("R") != std::string::npos
              || token.find("r") != std::string::npos)
          {
            plate = 0;
          }
          if (token.find("L") != std::string::npos
              || token.find("l") != std::string::npos)
          {
            plate = 1;
          }

          if (token.find("px") != std::string::npos)
          {
            cop = 0;
          }
          if (token.find("py") != std::string::npos)
          {
            cop = 1;
          }
          if (token.find("pz") != std::string::npos)
          {
            cop = 2;
          }
          if (token.find("mx") != std::string::npos)
          {
            wrench = 0;
          }
          if (token.find("my") != std::string::npos)
          {
            wrench = 1;
          }
          if (token.find("mz") != std::string::npos)
          {
            wrench = 2;
          }
          if (token.find("vx") != std::string::npos)
          {
            wrench = 3;
          }
          if (token.find("vy") != std::string::npos)
          {
            wrench = 4;
          }
          if (token.find("vz") != std::string::npos)
          {
            wrench = 5;
          }

          if (plate + 1 > numPlates)
          {
            numPlates = plate + 1;
          }

          colToPlate.push_back(plate);
          colToCOP.push_back(cop);
          colToWrench.push_back(wrench);
        }
        else
        {
          double value = atof(token.c_str());
          if (tokenNumber == 0)
          {
            timestamp = value;
          }
          else
          {
            int plateIndex = colToPlate[tokenNumber];
            int copIndex = colToCOP[tokenNumber];
            int wrenchIndex = colToWrench[tokenNumber];
            if (plateIndex != -1)
            {
              if (wrenchIndex != -1)
              {
                wrenches[plateIndex](wrenchIndex) = value;
              }
              if (copIndex != -1)
              {
                cops[plateIndex](copIndex) = value;
              }
            }
          }
        }

        /////////////////////////////////////////////////////////

        tokenNumber++;
        if (tokenEnd == std::string::npos)
        {
          break;
        }
        tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
      }

      if (lineNumber > 0)
      {
        copRows.push_back(cops);
        wrenchRows.push_back(wrenches);
        timestamps.push_back(timestamp);
      }
      lineNumber++;
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }

  assert(timestamps.size() == copRows.size());
  assert(timestamps.size() == wrenchRows.size());

  // Process result into its final form

  int numTimesteps = (int)ceil((double)timestamps.size() / downsampleByFactor);

  OpenSimGRF grf;
  for (int i = 0; i < numPlates; i++)
  {
    grf.plateCOPs.push_back(
        Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, numTimesteps));
    grf.plateGRFs.push_back(
        Eigen::Matrix<s_t, 6, Eigen::Dynamic>::Zero(6, numTimesteps));

    int downsampleClock = 0;
    Eigen::Vector3s copAvg = Eigen::Vector3s::Zero();
    Eigen::Vector6s wrenchAvg = Eigen::Vector6s::Zero();
    int numAveraged = 0;
    int cursor = 0;
    for (int t = 0; t < timestamps.size(); t++)
    {
      copAvg += copRows[t][i];
      wrenchAvg += wrenchRows[t][i];
      numAveraged++;
      downsampleClock--;

      if (downsampleClock <= 0)
      {
        downsampleClock = downsampleByFactor;
        grf.plateCOPs[i].col(cursor) = copAvg / numAveraged;
        grf.plateGRFs[i].col(cursor) = wrenchAvg / numAveraged;
        cursor++;

        numAveraged = 0;
        copAvg.setZero();
        wrenchAvg.setZero();
      }
    }
  }

  int downsampleClock = 0;
  for (int t = 0; t < timestamps.size(); t++)
  {
    downsampleClock--;
    if (downsampleClock <= 0)
    {
      grf.timestamps.push_back(timestamps[t]);
      downsampleClock = downsampleByFactor;
    }
  }

  return grf;
}

//==============================================================================
/// When people finish preparing their model in OpenSim, they save a *.osim
/// file with all the scales and offsets baked in. This is a utility to go
/// through and get out the scales and offsets in terms of a standard
/// skeleton, so that we can include their values in standard datasets.
OpenSimScaleAndMarkerOffsets OpenSimParser::getScaleAndMarkerOffsets(
    const OpenSimFile& standardSkeleton, const OpenSimFile& scaledSkeleton)
{
  OpenSimScaleAndMarkerOffsets config;

  // Check that the two skeletons are compatible
  for (int i = 0; i < standardSkeleton.skeleton->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* bodyNode = standardSkeleton.skeleton->getBodyNode(i);
    if (!scaledSkeleton.skeleton->getBodyNode(bodyNode->getName()))
    {
      std::cout << "OpenSimParser::getConfiguration() failed because the "
                   "skeletons were too different! The standard skeleton has a "
                   "body named \""
                << bodyNode->getName() << "\", but the scaled skeleton doesn't."
                << std::endl;
      config.success = false;
      return config;
    }
  }

  // Now go through both skeletons and try to work out what the body scales are

  config.bodyScales
      = Eigen::VectorXs::Ones(standardSkeleton.skeleton->getNumBodyNodes() * 3);
  for (int i = 0; i < standardSkeleton.skeleton->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* scaledNode = scaledSkeleton.skeleton->getBodyNode(
        standardSkeleton.skeleton->getBodyNode(i)->getName());
    dynamics::Shape* shape = scaledNode->getShapeNode(0)->getShape().get();
    if (shape->getType() == dynamics::MeshShape::getStaticType())
    {
      dynamics::MeshShape* mesh = static_cast<dynamics::MeshShape*>(shape);
      Eigen::Vector3s scale = mesh->getScale();
      config.bodyScales.segment<3>(i * 3) = scale;
    }
  }

  // Try to convert the markers into the standard skeleton

  for (auto pair : scaledSkeleton.markersMap)
  {
    std::string markerName = pair.first;
    dynamics::BodyNode* standardBody
        = standardSkeleton.skeleton->getBodyNode(pair.second.first->getName());
    if (standardBody != nullptr)
    {
      Eigen::Vector3s bodyScale = config.bodyScales.segment<3>(
          standardBody->getIndexInSkeleton() * 3);
      Eigen::Vector3s goldOffset = pair.second.second.cwiseQuotient(bodyScale);
      config.markers[markerName] = std::make_pair(standardBody, goldOffset);
      config.markerOffsets[markerName] = goldOffset - pair.second.second;
    }
  }

  // Return

  config.success = true;
  return config;
}

//==============================================================================
std::pair<dynamics::Joint*, dynamics::BodyNode*> createJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    tinyxml2::XMLElement* bodyCursor,
    tinyxml2::XMLElement* jointDetail,
    Eigen::Isometry3s transformFromParent,
    Eigen::Isometry3s transformFromChild,
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever)
{
  std::string bodyName(bodyCursor->Attribute("name"));
  dynamics::BodyNode::Properties bodyProps;
  bodyProps.mName = bodyName;

  dynamics::BodyNode* childBody = nullptr;
  std::string jointName(jointDetail->Attribute("name"));

  dynamics::Joint* joint = nullptr;
  std::string jointType(jointDetail->Name());

  // Build custom joints
  if (jointType == "CustomJoint")
  {
    tinyxml2::XMLElement* spatialTransform
        = jointDetail->FirstChildElement("SpatialTransform");
    tinyxml2::XMLElement* transformAxisCursor
        = spatialTransform->FirstChildElement("TransformAxis");

    std::vector<std::shared_ptr<math::CustomFunction>> customFunctions;
    std::vector<Eigen::Vector3s> eulerAxisOrder;
    std::vector<Eigen::Vector3s> transformAxisOrder;

    int dofIndex = 0;
    /// If all linear, then we're just a EulerFreeJoint
    bool allLinear = true;
    /// If first 1 is linear, last 5 are constant, then we're just a
    /// RevoluteJoint
    bool first1Linear = true;
    /// If first 3 are linear, last 3 are constant, then we're just an
    /// EulerJoint
    bool first3Linear = true;
    /// If any are splines, we need a full blown CustomJoint
    bool anySpline = false;
    /// If all 6 DOFs are fixed at 0, then we're just a WeldJoint
    bool allLocked = true;
    while (transformAxisCursor)
    {
      Eigen::Vector3s axis
          = readVec3(transformAxisCursor->FirstChildElement("axis"));

      tinyxml2::XMLElement* function
          = transformAxisCursor->FirstChildElement("function");
      // On v3 files, there is no "function" wrapper tag
      if (function == nullptr)
      {
        function = transformAxisCursor;
      }

      tinyxml2::XMLElement* linearFunction
          = function->FirstChildElement("LinearFunction");
      tinyxml2::XMLElement* simmSpline
          = function->FirstChildElement("SimmSpline");
      tinyxml2::XMLElement* polynomialFunction
          = function->FirstChildElement("PolynomialFunction");
      // This only exists in v4 files
      tinyxml2::XMLElement* constant = function->FirstChildElement("Constant");
      // This only exists in v3 files
      tinyxml2::XMLElement* multiplier
          = function->FirstChildElement("MultiplierFunction");
      if (multiplier != nullptr)
      {
        tinyxml2::XMLElement* childFunction
            = multiplier->FirstChildElement("function");
        assert(childFunction != nullptr);
        if (childFunction != nullptr)
        {
          constant = childFunction->FirstChildElement("Constant");
          simmSpline = childFunction->FirstChildElement("SimmSpline");
          linearFunction = childFunction->FirstChildElement("LinearFunction");
          polynomialFunction
              = childFunction->FirstChildElement("PolynomialFunction");
          assert(
              constant || simmSpline || linearFunction || polynomialFunction);
        }
      }

      if (constant != nullptr)
      {
        allLinear = false;
        if (dofIndex == 0)
        {
          first1Linear = false;
        }
        if (dofIndex < 3)
        {
          first3Linear = false;
        }

        double value = atof(constant->FirstChildElement("value")->GetText());
        if (value != 0)
        {
          allLocked = false;
        }
        customFunctions.push_back(
            std::make_shared<math::ConstantFunction>(value));
      }
      else if (linearFunction != nullptr)
      {
        allLocked = false;
        Eigen::Vector2s coeffs
            = readVec2(linearFunction->FirstChildElement("coefficients"));
        // Bake coeff flips into the axis
        if (coeffs(0) == -1)
        {
          axis *= coeffs(0);
          coeffs(0) = 1.0;
        }
        // Example coeffs for linear: 1 0
        customFunctions.push_back(
            std::make_shared<math::LinearFunction>(coeffs(0), coeffs(1)));
      }
      else if (polynomialFunction != nullptr)
      {
        anySpline = true;
        allLocked = false;
        std::vector<s_t> coeffs
            = readVecX(polynomialFunction->FirstChildElement("coefficients"));
        customFunctions.push_back(
            std::make_shared<math::PolynomialFunction>(coeffs));
      }
      else if (simmSpline != nullptr)
      {
        anySpline = true;
        allLocked = false;
        if (dofIndex == 0)
        {
          first1Linear = false;
        }
        if (dofIndex < 3)
        {
          first3Linear = false;
        }

        std::vector<s_t> x = readVecX(simmSpline->FirstChildElement("x"));
        std::vector<s_t> y = readVecX(simmSpline->FirstChildElement("y"));
        customFunctions.push_back(std::make_shared<math::SimmSpline>(x, y));
      }
      else
      {
        assert(false && "Unrecognized function type");
      }

      if (dofIndex < 3)
        eulerAxisOrder.push_back(axis);
      else
        transformAxisOrder.push_back(axis);

      dofIndex++;
      transformAxisCursor
          = transformAxisCursor->NextSiblingElement("TransformAxis");
    }

    if (allLocked)
    {
      dynamics::WeldJoint::Properties props;
      props.mName = jointName;
      auto pair
          = parentBody->createChildJointAndBodyNodePair<dynamics::WeldJoint>(
              props, bodyProps);
      joint = pair.first;
      childBody = pair.second;
      std::cout << "WARNING! Creating a WeldJoint as an intermediate "
                   "(non-root) joint, there is a known bug that will cause "
                   "dynamics to be wrong. This is only useful for kinematics."
                << std::endl;
    }
    else if (anySpline)
    {
      dynamics::CustomJoint* customJoint = nullptr;
      // Create a CustomJoint
      dynamics::CustomJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::CustomJoint>(
            nullptr, props, bodyProps);
        customJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody
                  ->createChildJointAndBodyNodePair<dynamics::CustomJoint>(
                      props, bodyProps);
        customJoint = pair.first;
        childBody = pair.second;
      }

      assert(customFunctions.size() == 6);

      dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
      Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);
      customJoint->setAxisOrder(axisOrder);
      customJoint->setFlipAxisMap(flips);

      for (int i = 0; i < customFunctions.size(); i++)
      {
        if (i < 3)
        {
          customJoint->setCustomFunction(i, customFunctions[i]);
        }
        else
        {
          // Map to the appropriate slot based on the axis
          Eigen::Vector3s axis = transformAxisOrder[i - 3];
          if (axis == Eigen::Vector3s::UnitX())
          {
            customJoint->setCustomFunction(3, customFunctions[i]);
          }
          else if (axis == Eigen::Vector3s::UnitY())
          {
            customJoint->setCustomFunction(4, customFunctions[i]);
          }
          else if (axis == Eigen::Vector3s::UnitZ())
          {
            customJoint->setCustomFunction(5, customFunctions[i]);
          }
          else
          {
            assert(false);
          }
        }
      }
      joint = customJoint;
    }
    else if (allLinear)
    {
      dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
      dynamics::EulerJoint::AxisOrder transOrder
          = getAxisOrder(transformAxisOrder);
      (void)transOrder;
      assert(transOrder == dynamics::EulerJoint::AxisOrder::XYZ);

      Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);

      // Create a EulerFreeJoint
      dynamics::EulerFreeJoint* eulerFreeJoint = nullptr;
      dynamics::EulerFreeJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>(
            nullptr, props, bodyProps);
        eulerFreeJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody
                  ->createChildJointAndBodyNodePair<dynamics::EulerFreeJoint>(
                      props, bodyProps);
        eulerFreeJoint = pair.first;
        childBody = pair.second;
      }
      eulerFreeJoint->setAxisOrder(axisOrder);
      eulerFreeJoint->setFlipAxisMap(flips);
      joint = eulerFreeJoint;
    }
    else if (first3Linear)
    {
      dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
      Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);
      // assert(!flips[0] && !flips[1] && !flips[2]);

      // Create an EulerJoint
      dynamics::EulerJoint* eulerJoint = nullptr;
      dynamics::EulerJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::EulerJoint>(
            nullptr, props, bodyProps);
        eulerJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody->createChildJointAndBodyNodePair<dynamics::EulerJoint>(
                props, bodyProps);
        eulerJoint = pair.first;
        childBody = pair.second;
      }
      eulerJoint->setFlipAxisMap(flips);
      eulerJoint->setAxisOrder(axisOrder);
      joint = eulerJoint;
    }
    else if (first1Linear)
    {
      Eigen::Vector3s axis = eulerAxisOrder[0];

      // Create a RevoluteJoint
      dynamics::RevoluteJoint* revoluteJoint = nullptr;
      dynamics::RevoluteJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::RevoluteJoint>(
            nullptr, props, bodyProps);
        revoluteJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody
                  ->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>(
                      props, bodyProps);
        revoluteJoint = pair.first;
        childBody = pair.second;
      }
      revoluteJoint->setAxis(axis);
      joint = revoluteJoint;
    }
    else
    {
      assert(false);
    }
  }
  if (jointType == "WeldJoint")
  {
    dynamics::WeldJoint::Properties props;
    props.mName = jointName;
    auto pair
        = parentBody->createChildJointAndBodyNodePair<dynamics::WeldJoint>(
            props, bodyProps);
    joint = pair.first;
    childBody = pair.second;
    std::cout << "WARNING! Creating a WeldJoint as an intermediate "
                 "(non-root) joint, there is a known bug that will cause "
                 "dynamics to be wrong. This is only useful for kinematics."
              << std::endl;
  }
  if (jointType == "PinJoint")
  {
    // Create a RevoluteJoint
    dynamics::RevoluteJoint* revoluteJoint = nullptr;
    dynamics::RevoluteJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<dynamics::RevoluteJoint>(
          nullptr, props, bodyProps);
      revoluteJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair
          = parentBody
                ->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>(
                    props, bodyProps);
      revoluteJoint = pair.first;
      childBody = pair.second;
    }
    joint = revoluteJoint;
  }
  if (jointType == "UniversalJoint")
  {
    // Create a UniversalJoint
    dynamics::UniversalJoint* universalJoint = nullptr;
    dynamics::UniversalJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<dynamics::UniversalJoint>(
          nullptr, props, bodyProps);
      universalJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair
          = parentBody
                ->createChildJointAndBodyNodePair<dynamics::UniversalJoint>(
                    props, bodyProps);
      universalJoint = pair.first;
      childBody = pair.second;
    }
    joint = universalJoint;
  }
  assert(childBody != nullptr);
  joint->setName(jointName);
  // std::cout << jointName << std::endl;
  joint->setTransformFromChildBodyNode(transformFromChild);
  joint->setTransformFromParentBodyNode(transformFromParent);

  // Rename the DOFs for each joint

  tinyxml2::XMLElement* coordinateCursor = nullptr;
  // This is how the coordinate set is specified in OpenSim v4 files
  tinyxml2::XMLElement* coordinateSet
      = jointDetail->FirstChildElement("CoordinateSet");
  if (coordinateSet)
  {
    tinyxml2::XMLElement* objects = coordinateSet->FirstChildElement("objects");
    if (objects)
    {
      coordinateCursor = objects->FirstChildElement("Coordinate");
    }
  }
  // This is how the coordinate set is specified in OpenSim v3 files
  if (coordinateCursor == nullptr)
  {
    coordinateSet = jointDetail->FirstChildElement("coordinates");
    if (coordinateSet != nullptr)
    {
      coordinateCursor = coordinateSet->FirstChildElement("Coordinate");
    }
  }
  // Iterate through the coordinates
  int i = 0;
  while (coordinateCursor)
  {
    std::string dofName(coordinateCursor->Attribute("name"));
    double defaultValue
        = atof(coordinateCursor->FirstChildElement("default_value")->GetText());
    double defaultSpeedValue = atof(
        coordinateCursor->FirstChildElement("default_speed_value")->GetText());
    Eigen::Vector2s range
        = readVec2(coordinateCursor->FirstChildElement("range"));
    bool locked
        = std::string(coordinateCursor->FirstChildElement("locked")->GetText())
          == "true";
    bool clamped
        = std::string(coordinateCursor->FirstChildElement("clamped")->GetText())
          == "true";

    dynamics::DegreeOfFreedom* dof = joint->getDof(i);
    dof->setName(dofName);
    dof->setPosition(defaultValue);
    dof->setVelocity(defaultSpeedValue);
    if (locked)
    {
      // TODO: Just replace with a Weld joint
      // dof->setVelocityUpperLimit(0);
      // dof->setVelocityLowerLimit(0);
    }
    if (clamped)
    {
      dof->setPositionLowerLimit(range(0));
      dof->setPositionUpperLimit(range(1));
    }

    i++;
    coordinateCursor = coordinateCursor->NextSiblingElement("Coordinate");
  }

  // OpenSim v4 files specify visible geometry this way
  tinyxml2::XMLElement* visibleObject
      = bodyCursor->FirstChildElement("VisibleObject");
  if (visibleObject && childBody != nullptr)
  {
    tinyxml2::XMLElement* geometrySet
        = visibleObject->FirstChildElement("GeometrySet");
    if (geometrySet)
    {
      tinyxml2::XMLElement* objects = geometrySet->FirstChildElement("objects");
      if (objects)
      {
        tinyxml2::XMLElement* displayGeometryCursor
            = objects->FirstChildElement("DisplayGeometry");
        while (displayGeometryCursor)
        {
          std::string mesh_file(
              displayGeometryCursor->FirstChildElement("geometry_file")
                  ->GetText());
          Eigen::Vector3s colors
              = readVec3(displayGeometryCursor->FirstChildElement("color"))
                * 0.7;
          Eigen::Vector6s transformVec
              = readVec6(displayGeometryCursor->FirstChildElement("transform"));
          Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
          transform.linear() = math::eulerXYZToMatrix(transformVec.head<3>());
          transform.translation() = transformVec.tail<3>();
          Eigen::Vector3s scale = readVec3(
              displayGeometryCursor->FirstChildElement("scale_factors"));
          double opacity = atof(
              displayGeometryCursor->FirstChildElement("opacity")->GetText());

          common::Uri meshUri = common::Uri::createFromRelativeUri(
              uri, "./Geometry/" + mesh_file + ".ply");
          std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr
              = dynamics::MeshShape::loadMesh(meshUri, retriever);

          if (meshPtr)
          {
            std::shared_ptr<dynamics::MeshShape> meshShape
                = std::make_shared<dynamics::MeshShape>(
                    scale, meshPtr, meshUri, retriever);

            dynamics::ShapeNode* meshShapeNode
                = childBody->createShapeNodeWith<dynamics::VisualAspect>(
                    meshShape);
            meshShapeNode->setRelativeTransform(transform);

            dynamics::VisualAspect* meshVisualAspect
                = meshShapeNode->getVisualAspect();
            meshVisualAspect->setColor(colors);
            meshVisualAspect->setAlpha(opacity);
          }

          displayGeometryCursor
              = displayGeometryCursor->NextSiblingElement("DisplayGeometry");
        }
      }
    }
  }
  // OpenSim v3 files specify visible geometry this way
  tinyxml2::XMLElement* attachedGeometry
      = bodyCursor->FirstChildElement("attached_geometry");
  if (attachedGeometry && childBody != nullptr)
  {
    tinyxml2::XMLElement* meshCursor
        = attachedGeometry->FirstChildElement("Mesh");
    while (meshCursor)
    {
      std::string mesh_file(
          meshCursor->FirstChildElement("mesh_file")->GetText());
      Eigen::Vector3s scale
          = readVec3(meshCursor->FirstChildElement("scale_factors"));

      common::Uri meshUri = common::Uri::createFromRelativeUri(
          uri, "./Geometry/" + mesh_file + ".ply");
      std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr
          = dynamics::MeshShape::loadMesh(meshUri, retriever);

      if (meshPtr)
      {
        std::shared_ptr<dynamics::MeshShape> meshShape
            = std::make_shared<dynamics::MeshShape>(
                scale, meshPtr, meshUri, retriever);

        dynamics::ShapeNode* meshShapeNode
            = childBody->createShapeNodeWith<dynamics::VisualAspect>(meshShape);

        /*
        Eigen::Vector6s transformVec
            = readVec6(displayGeometryCursor->FirstChildElement("transform"));
        Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
        transform.linear() = math::eulerXYZToMatrix(transformVec.head<3>());
        transform.translation() = transformVec.tail<3>();
        meshShapeNode->setRelativeTransform(transform);
        */

        dynamics::VisualAspect* meshVisualAspect
            = meshShapeNode->getVisualAspect();

        tinyxml2::XMLElement* appearance
            = meshCursor->FirstChildElement("Appearance");
        if (appearance != nullptr)
        {
          Eigen::Vector3s colors
              = readVec3(appearance->FirstChildElement("color")) * 0.7;
          double opacity
              = atof(appearance->FirstChildElement("opacity")->GetText());
          meshVisualAspect->setColor(colors);
          meshVisualAspect->setAlpha(opacity);
        }
      }

      meshCursor = meshCursor->NextSiblingElement("Mesh");
    }
  }

  assert(childBody != nullptr);

  return std::pair<dynamics::Joint*, dynamics::BodyNode*>(joint, childBody);
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim40(
    const common::Uri& uri,
    tinyxml2::XMLElement* docElement,
    const common::ResourceRetrieverPtr& retriever)
{
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");

  if (bodySet == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] missing <BodySet> group.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  //--------------------------------------------------------------------------
  // Build out the physical structure
  unordered_map<string, dynamics::BodyNode*> bodyLookupMap;
  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    std::string name(bodyCursor->Attribute("name"));
    // std::cout << name << std::endl;

    if (name == "ground")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }
    // Skip the kneecaps
    if (name == "patella_r" || name == "patella_l")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }

    tinyxml2::XMLElement* joint = bodyCursor->FirstChildElement("Joint");

    dynamics::BodyNode* childBody = nullptr;

    if (joint)
    {
      // WeldJoint
      if (joint->FirstChild() == nullptr)
      {
      }
      tinyxml2::XMLElement* jointDetail = nullptr;

      // CustomJoint
      tinyxml2::XMLElement* customJoint
          = joint->FirstChildElement("CustomJoint");
      if (customJoint)
      {
        jointDetail = customJoint;
      }
      // PinJoint
      tinyxml2::XMLElement* pinJoint = joint->FirstChildElement("PinJoint");
      if (pinJoint)
      {
        jointDetail = pinJoint;
      }
      // UniversalJoint
      tinyxml2::XMLElement* universalJoint
          = joint->FirstChildElement("UniversalJoint");
      if (universalJoint)
      {
        jointDetail = universalJoint;
      }

      if (jointDetail != nullptr)
      {
        std::string parentName = std::string(
            jointDetail->FirstChildElement("parent_body")->GetText());
        dynamics::BodyNode* parentBody = bodyLookupMap[parentName];
        assert(parentName == "ground" || parentBody != nullptr);
        // Get shared properties across all joint types
        Eigen::Vector3s locationInParent
            = readVec3(jointDetail->FirstChildElement("location_in_parent"));
        Eigen::Vector3s orientationInParent
            = readVec3(jointDetail->FirstChildElement("orientation_in_parent"));

        Eigen::Vector3s locationInChild
            = readVec3(jointDetail->FirstChildElement("location"));
        Eigen::Vector3s orientationInChild
            = readVec3(jointDetail->FirstChildElement("orientation"));

        Eigen::Isometry3s transformFromParent = Eigen::Isometry3s::Identity();
        transformFromParent.linear()
            = math::eulerXYZToMatrix(orientationInParent);
        transformFromParent.translation() = locationInParent;
        Eigen::Isometry3s transformFromChild = Eigen::Isometry3s::Identity();
        transformFromChild.linear()
            = math::eulerXYZToMatrix(orientationInChild);
        transformFromChild.translation() = locationInChild;

        auto pair = createJoint(
            skel,
            parentBody,
            bodyCursor,
            jointDetail,
            transformFromParent,
            transformFromChild,
            uri,
            retriever);
        childBody = pair.second;
      }
    }
    double mass = atof(bodyCursor->FirstChildElement("mass")->GetText());
    Eigen::Vector3s massCenter
        = readVec3(bodyCursor->FirstChildElement("mass_center"));
    double inertia_xx
        = atof(bodyCursor->FirstChildElement("inertia_xx")->GetText());
    double inertia_yy
        = atof(bodyCursor->FirstChildElement("inertia_yy")->GetText());
    double inertia_zz
        = atof(bodyCursor->FirstChildElement("inertia_zz")->GetText());
    double inertia_xy
        = atof(bodyCursor->FirstChildElement("inertia_xy")->GetText());
    double inertia_xz
        = atof(bodyCursor->FirstChildElement("inertia_xz")->GetText());
    double inertia_yz
        = atof(bodyCursor->FirstChildElement("inertia_yz")->GetText());
    dynamics::Inertia inertia(
        mass,
        massCenter(0),
        massCenter(1),
        massCenter(2),
        inertia_xx,
        inertia_yy,
        inertia_zz,
        inertia_xy,
        inertia_xz,
        inertia_yz);
    childBody->setInertia(inertia);

    childBody->setName(name);
    bodyLookupMap[name] = childBody;

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  /*
  std::cout << "Num dofs: " << skel->getNumDofs() << std::endl;
  std::cout << "Num bodies: " << skel->getNumBodyNodes() << std::endl;
  */

  OpenSimFile file;
  file.skeleton = skel;

  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");
  if (markerSet != nullptr)
  {
    tinyxml2::XMLElement* markerList = markerSet->FirstChildElement("objects");
    if (markerList != nullptr)
    {
      tinyxml2::XMLElement* markerCursor
          = markerList->FirstChildElement("Marker");
      while (markerCursor)
      {
        std::string name(markerCursor->Attribute("name"));
        Eigen::Vector3s offset
            = readVec3(markerCursor->FirstChildElement("location"));
        std::string bodyName
            = markerCursor->FirstChildElement("body")->GetText();
        bool fixed
            = std::string(markerCursor->FirstChildElement("fixed")->GetText())
              == "true";

        file.markersMap[name]
            = std::make_pair(skel->getBodyNode(bodyName), offset);
        if (fixed)
        {
          file.anatomicalMarkers.push_back(name);
        }
        else
        {
          file.trackingMarkers.push_back(name);
        }

        markerCursor = markerCursor->NextSiblingElement();
      }
    }
  }

  return file;
}

struct OpenSimBodyXML;

struct OpenSimJointXML
{
  string name;
  string type;
  OpenSimBodyXML* parent;
  OpenSimBodyXML* child;
  Eigen::Isometry3s fromParent;
  Eigen::Isometry3s fromChild;
  tinyxml2::XMLElement* xml;
  dynamics::BodyNode* parentBody;
};

struct OpenSimBodyXML
{
  string name;
  OpenSimJointXML* parent;
  std::vector<OpenSimJointXML*> children;
  tinyxml2::XMLElement* xml;
};

//==============================================================================
void recursiveCreateJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    OpenSimJointXML* joint,
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever)
{
  (void)skel;
  (void)parentBody;
  (void)joint;

  // std::cout << "Building joint: " << joint->name << std::endl;

  tinyxml2::XMLElement* jointNode = joint->xml;
  tinyxml2::XMLElement* bodyNode = joint->child->xml;

  auto pair = createJoint(
      skel,
      parentBody,
      bodyNode,
      jointNode,
      joint->fromParent,
      joint->fromChild,
      uri,
      retriever);

  dynamics::BodyNode* childBody = pair.second;

  double mass = atof(bodyNode->FirstChildElement("mass")->GetText());
  Eigen::Vector3s massCenter
      = readVec3(bodyNode->FirstChildElement("mass_center"));
  Eigen::Vector6s inertia = readVec6(bodyNode->FirstChildElement("inertia"));
  dynamics::Inertia inertiaObj(
      mass,
      massCenter(0),
      massCenter(1),
      massCenter(2),
      inertia(0),
      inertia(1),
      inertia(2),
      inertia(3),
      inertia(4),
      inertia(5));
  childBody->setInertia(inertiaObj);

  // Recurse to the next layer
  for (OpenSimJointXML* grandChildJoint : joint->child->children)
  {
    recursiveCreateJoint(skel, childBody, grandChildJoint, uri, retriever);
  }
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim30(
    const common::Uri& uri,
    tinyxml2::XMLElement* docElement,
    const common::ResourceRetrieverPtr& retriever)
{
  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  (void)retriever;
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (bodySet == nullptr || jointSet == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] missing <BodySet> or <JointSet> groups.\n";
    return null_file;
  }

  //--------------------------------------------------------------------------
  // Read BodySet
  unordered_map<string, OpenSimBodyXML> bodyLookupMap;

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    std::string name(bodyCursor->Attribute("name"));
    // std::cout << name << std::endl;

    bodyLookupMap["/bodyset/" + name].xml = bodyCursor;
    bodyLookupMap["/bodyset/" + name].parent = nullptr;
    bodyLookupMap["/bodyset/" + name].children.clear();
    bodyLookupMap["/bodyset/" + name].name = name;

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Read JointSet
  unordered_map<string, OpenSimJointXML> jointLookupMap;

  tinyxml2::XMLElement* jointSetList = jointSet->FirstChildElement("objects");
  tinyxml2::XMLElement* jointCursor = jointSetList->FirstChildElement();
  while (jointCursor)
  {
    std::string type(jointCursor->Name());
    std::string name(jointCursor->Attribute("name"));

    string parent_offset_frame = string(
        jointCursor->FirstChildElement("socket_parent_frame")->GetText());
    string child_offset_frame = string(
        jointCursor->FirstChildElement("socket_child_frame")->GetText());

    tinyxml2::XMLElement* frames = jointCursor->FirstChildElement("frames");
    tinyxml2::XMLElement* framesCursor = frames->FirstChildElement();

    string parentName = "";
    Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
    string childName = "";
    Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();

    while (framesCursor)
    {
      string parent_body
          = string(framesCursor->FirstChildElement("socket_parent")->GetText());
      Eigen::Vector3s translation
          = readVec3(framesCursor->FirstChildElement("translation"));
      Eigen::Vector3s rotationXYZ
          = readVec3(framesCursor->FirstChildElement("orientation"));
      Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
      T.linear() = math::eulerXYZToMatrix(rotationXYZ);
      T.translation() = translation;

      string name(framesCursor->Attribute("name"));
      if (name == parent_offset_frame)
      {
        parentName = parent_body;
        fromParent = T;
      }
      else if (name == child_offset_frame)
      {
        childName = parent_body;
        fromChild = T;
      }

      framesCursor = framesCursor->NextSiblingElement();
    }

    OpenSimJointXML& joint = jointLookupMap[name];
    joint.name = name;
    joint.type = type;
    joint.fromParent = fromParent;
    joint.fromChild = fromChild;
    joint.xml = jointCursor;

    // A body can have only one parent joint
    if (bodyLookupMap.find(parentName) != bodyLookupMap.end())
    {
      bodyLookupMap[parentName].children.push_back(&joint);
      joint.parent = &bodyLookupMap[parentName];
    }
    else
    {
      joint.parent = nullptr;
    }

    assert(bodyLookupMap.find(childName) != bodyLookupMap.end());
    assert(bodyLookupMap[childName].parent == nullptr);
    joint.child = &bodyLookupMap[childName];
    bodyLookupMap[childName].parent = &joint;

    jointCursor = jointCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Check tree invarients

  OpenSimJointXML* root = nullptr;
  for (auto& pair : jointLookupMap)
  {
    if (pair.second.parent == nullptr)
    {
      assert(root == nullptr);
      root = &pair.second;
    }
  }

  assert(root != nullptr);

  //--------------------------------------------------------------------------
  // Build out the physical structure

  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();
  root->parentBody = nullptr;
  recursiveCreateJoint(skel, nullptr, root, uri, retriever);

  /*
  std::cout << "Num dofs: " << skel->getNumDofs() << std::endl;
  std::cout << "Num bodies: " << skel->getNumBodyNodes() << std::endl;
  */

  OpenSimFile file;
  file.skeleton = skel;

  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");
  if (markerSet != nullptr)
  {
    tinyxml2::XMLElement* markerList = markerSet->FirstChildElement("objects");
    if (markerList != nullptr)
    {
      tinyxml2::XMLElement* markerCursor
          = markerList->FirstChildElement("Marker");
      while (markerCursor)
      {
        std::string name(markerCursor->Attribute("name"));
        Eigen::Vector3s offset
            = readVec3(markerCursor->FirstChildElement("location"));
        std::string socketName
            = markerCursor->FirstChildElement("socket_parent_frame")->GetText();
        std::string bodyName = bodyLookupMap[socketName].name;

        file.markersMap[name]
            = std::make_pair(skel->getBodyNode(bodyName), offset);

        markerCursor = markerCursor->NextSiblingElement();
      }
    }
  }

  return file;
}

}; // namespace biomechanics
}; // namespace dart